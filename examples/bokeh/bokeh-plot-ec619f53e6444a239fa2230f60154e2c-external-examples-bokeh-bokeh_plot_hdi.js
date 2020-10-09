(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("d1d37b4b-1970-47fb-aff5-3aca09453b08");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'd1d37b4b-1970-47fb-aff5-3aca09453b08' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"8b508dfa-8af8-4aad-92c8-2aed72e4c30b":{"roots":{"references":[{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5288"},"selection_policy":{"id":"5289"}},"id":"5272","type":"ColumnDataSource"},{"attributes":{},"id":"5284","type":"BasicTickFormatter"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5279","type":"Line"},{"attributes":{},"id":"5291","type":"UnionRenderers"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5261","type":"BoxAnnotation"},{"attributes":{"source":{"id":"5277"}},"id":"5281","type":"CDSView"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5253"},{"id":"5254"},{"id":"5255"},{"id":"5256"},{"id":"5257"},{"id":"5258"},{"id":"5259"},{"id":"5260"}]},"id":"5263","type":"Toolbar"},{"attributes":{"callback":null},"id":"5260","type":"HoverTool"},{"attributes":{"data_source":{"id":"5277"},"glyph":{"id":"5278"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5279"},"selection_glyph":null,"view":{"id":"5281"}},"id":"5280","type":"GlyphRenderer"},{"attributes":{},"id":"5246","type":"BasicTicker"},{"attributes":{},"id":"5256","type":"WheelZoomTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5262","type":"PolyAnnotation"},{"attributes":{"formatter":{"id":"5284"},"ticker":{"id":"5246"}},"id":"5245","type":"LinearAxis"},{"attributes":{},"id":"5259","type":"SaveTool"},{"attributes":{},"id":"5253","type":"ResetTool"},{"attributes":{"axis":{"id":"5245"},"ticker":null},"id":"5248","type":"Grid"},{"attributes":{},"id":"5254","type":"PanTool"},{"attributes":{},"id":"5243","type":"LinearScale"},{"attributes":{"data":{"x":{"__ndarray__":"3xqCoWV8B8BGKeExfusCwLkii/f4JgHAE9ExlLBVAMAOItBTIMf8vyCoxPoWFfm/VEzoNq0H+L9q+LEyGKr0vy7AHLOiGPS/Kh96pAT3878k1oZ4Adrzv8hjLI5RkvO/Qp5kTyvz8r8qTF8jBD7yv9UouWGfHfK//07rQyoW8r/ra/mT6PDxv6S9EEDNXPG/DE0si4nw8L+oRhrYqGfwvw/Tow1cRO+/KtTJxgHR7r954VkJMrHuvz05Txv5ie6/FOv/ofXS7b/aWkuFa7Psv1zWXoeSZOy/OKbrw+Qp7L8e+M3cjc3rvzf0gX6Zt+u/V/z3uPvI6r8H+RKeBuTov2znAEnhbei/0PyFa8j+5r+S8mvrClTkvyTEPOuE0+O/xaA6BR/v4r9jDYrJtf/hv8SVmqv9TuG/EVrHEk324L+nSB+umKPfv8q2uUorJ9q/dKfm/guJ2b+Uyb18K0rZvyh1am+xR9i/UWD3A21U1r9t3pPqsQXWv16LZ7HVctS/lGz+up7K079EZKZaN/7Ov6NQOEUur8q/J60C5cvCwb8HwZvbYrrBv/Y7g48LqLy/FPIyg0X1ub9Hbwy2YSe2vwbaf54JnbG/52SKMRXfqb/aafXGP52Xv+jDPAwPEKM/XvGBBk5LtT/gtEBID3O6P7fONueXP70/TiHk+u31yT938bFwaqzQP78Rja0GVdI/2zUrs32i0j8nWc8NyHXYP5lf3npKiNg/mgJHw2rk2D9Err18YSbZP9MY+h/9DNs/Cp5SZnrd3D9h3sdUc4feP2aM5yPszN4/oMOBCxMQ4D/o/q7B1U3gPzWIAdmYQeI/lmB9NeYS4z8m038EOWHjP2UC8PxCEuQ/yeubNIof5D+IsUZ7oSvkP/3v85f9d+U/CuaXxWhW5j8Qql+5x27sP7ttYTKfdew/LmQABnYc8D9hkjGt4D3xPw5iC1aScvE/5e+wvb0H8j9SEGPJZxbyPx+dfFygH/I/8awnwcB68j9mXUwpMXfzP6/8mxUiVvk/vPl6TkqY/D+q5gJlhT8AQHmvSB90FAJADYYOO+yWBUA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"Qsr7vDQH4T90rT2cAynqP4666RAOsu0/2l2c155U7z/57hfWb5zxP/CrnYJ0dfM/1tmLZCn88z/LA6fm86r1P+mfcaau8/U/a/DCrX0E9j/ulLxD/xL2PxzO6TjXNvY/37BNWGqG9j/rWVDu/eD2P5ZrI08w8fY/gFgK3ur09j8KSgO2iwf3Py6h91+ZUfc/etlpOruH9z+s3PKTK8z3PzwLl/zoLvg/9opNjr9L+D+ih6l9s1P4P7ExLLmBXfg/OwWAl0KL+D9KKa0eJdP4P2lKKF7b5vg/chYFz4b1+D/4gcyInAz5P/KCX6AZEvk/6gDCEcFN+T++QXtY/sb5PyXGv62H5Pk/zIAe5U1A+j9cAyVF/er6P/fOMMUeC/s/z1exPjhE+z+nfJ2NEoD7P49aGZVArPs/fClOu2zC+z/rFjzqjAv8PyfJqJYau/w/EisjgN7O/D/ORmiQutb8P1uxEtIJ9/w/9hOBX3I1/T8yhK3CST/9P5QO00mlcf0/bjKgKKyG/T+8mVWKHBD+P/Z6rBsNVf4/LtWvQdPj/j/wQ0bSWeT+PyDmg6O/Gv8/b2jm01Uw/z+GnE/yxE7/PzABDLMXc/8/bNY5q4OY/z8sFXKAxdD/P4h5GB4gJgBAxQcaOC1VAEDUAiE9zGkAQDvbnF/+dABACiHXb6/PAEAXHwunxgoBQBzR2GpQJQFAXrMy2ycqAUCS9dyAXIcBQPrlraeEiAFAKnA0rEaOAUDk2ssXZpIBQI2h/9HPsAFA4SllptfNAUDmfUw1d+gBQMZ4PsLO7AFAdDhwYQICAkDd3zW4ugkCQAcxIBszSAJAE6yvxlxiAkBl+o8gJ2wCQE0Anl9IggJAeX2TRvGDAkAx1mgvdIUCQAB+/rL/rgJAwfyyGM3KAkBC9Sv32I0DQLctTOazjgNADBmAgR0HBECYZEwreE8EQITYgpWkXARA+Ttsb++BBEAUxFjymYUEQEgnHxfohwRAPOtJMLCeBEBaF1NKzN0EQCz/ZoWIVQZAb76ekxImB0BVc4Gywh8IQLxXpA86CglABkOHHXbLCkA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5290"},"selection_policy":{"id":"5291"}},"id":"5277","type":"ColumnDataSource"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5278","type":"Line"},{"attributes":{"overlay":{"id":"5262"}},"id":"5257","type":"LassoSelectTool"},{"attributes":{},"id":"5288","type":"Selection"},{"attributes":{},"id":"5289","type":"UnionRenderers"},{"attributes":{"below":[{"id":"5245"}],"center":[{"id":"5248"},{"id":"5252"}],"left":[{"id":"5249"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5275"},{"id":"5280"}],"title":{"id":"5282"},"toolbar":{"id":"5263"},"toolbar_location":"above","x_range":{"id":"5237"},"x_scale":{"id":"5241"},"y_range":{"id":"5239"},"y_scale":{"id":"5243"}},"id":"5236","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"5272"}},"id":"5276","type":"CDSView"},{"attributes":{},"id":"5286","type":"BasicTickFormatter"},{"attributes":{},"id":"5239","type":"DataRange1d"},{"attributes":{"text":""},"id":"5282","type":"Title"},{"attributes":{},"id":"5241","type":"LinearScale"},{"attributes":{},"id":"5237","type":"DataRange1d"},{"attributes":{"data_source":{"id":"5272"},"glyph":{"id":"5273"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5274"},"selection_glyph":null,"view":{"id":"5276"}},"id":"5275","type":"GlyphRenderer"},{"attributes":{},"id":"5258","type":"UndoTool"},{"attributes":{},"id":"5250","type":"BasicTicker"},{"attributes":{"overlay":{"id":"5261"}},"id":"5255","type":"BoxZoomTool"},{"attributes":{"formatter":{"id":"5286"},"ticker":{"id":"5250"}},"id":"5249","type":"LinearAxis"},{"attributes":{},"id":"5290","type":"Selection"},{"attributes":{"axis":{"id":"5249"},"dimension":1,"ticker":null},"id":"5252","type":"Grid"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5273","type":"Patch"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5274","type":"Patch"}],"root_ids":["5236"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"8b508dfa-8af8-4aad-92c8-2aed72e4c30b","root_ids":["5236"],"roots":{"5236":"d1d37b4b-1970-47fb-aff5-3aca09453b08"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();