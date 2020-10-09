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
    
      
      
    
      var element = document.getElementById("27e2e92c-12c7-4e9b-bb58-875e9804fc1c");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '27e2e92c-12c7-4e9b-bb58-875e9804fc1c' but no matching script tag was found.")
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
                    
                  var docs_json = '{"96d93505-213d-47ae-ab30-96bde07684a9":{"roots":{"references":[{"attributes":{},"id":"5256","type":"WheelZoomTool"},{"attributes":{},"id":"5287","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"5277"}},"id":"5281","type":"CDSView"},{"attributes":{},"id":"5239","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5253"},{"id":"5254"},{"id":"5255"},{"id":"5256"},{"id":"5257"},{"id":"5258"},{"id":"5259"},{"id":"5260"}]},"id":"5263","type":"Toolbar"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5279","type":"Line"},{"attributes":{},"id":"5246","type":"BasicTicker"},{"attributes":{"data":{"x":{"__ndarray__":"qLfAcSQUC8B6b6xjon4CwF2UGW9qeADA7RKJhjZr/7/cjH3AzVL9vzIkz59qMPy/iexkM67++r+kGAZoyfv3vx93etyncfe/+7BYaSrq9b/xbZi1xHH1v6OfaSKQMvW/lkQ3HT7t9L+pNR8M13r0v0mQu2t0nfK/kNO3N89D8b/QPqLkVbPwv8rxe2teZ+6/W5nbAGh/679BK/QJm5Tpv7RLk2BXROm/moH0alCv57+BGSVCn57lv6j8CIYTLeW/9GroAtcV5b9ftN4/Dirgv19+ZdL5qdy/hiqDfjBy2r+/NjVt+MXXv/fjF6DF4da/n8jOlTkr07/DgsKfazbSv6gGnklXFM6/3h7a2CctyL+ZjA1irgrIvxIOEj3CzcW//np5pX4ZxL+OcvP/xTHDvxb4VieLOLu/2kVbhGGwur+KhPOt34q4v3ztK1O3tqe/pdIkncXOlr/zW8CdBMyNvx+u3kfiYYa/nN+MtHE4kD9h6U4QNwK7PweE5b4efL4/Dqm/OTkYxD/FEsJzfejEPxxgK0MrXMg/jH89vw3ayD+PC+UVd2TQP+0v28ZCydQ/Sp3xTfwf1T9rkuyhdzLYP959WP/HI9o/6jhSX4+H3z+p/wQUWN7fP41qhcQN9OA/y7HS7KMi4j9zRtX4I5biP1Is5DyB9OM/K81lnLj15T9qK+qihZjmP2nZmMwbdOc/AYLQnQsf6T9YiG2j/3vpP18XxBi7m+k/kXPKqfq76j97MWNB+xPrP0jR6dbeVus/lVGpNs2J6z9D8HNLEqHsP9OccVfUy+w/sKEtvhSe7z9Ff7leF33wP0tBCpDAn/E/SgC56v3P8T/qwWOGlwryP1TWlcQShPI/W+VYnWyo8j9XHs+nPSzzP2cYXNh8aPM//D23lxGV9T8OrJ/zpeD1P3xIHahNHvY/COG1rf859j+G7zFZm9f2P90clgJdH/c//v8oJP6d+D+r5GFmD4/7P0Ogau9z1vw/QF/hLapc/T+/jtd7qocAQBta5PjmEwFAyFByLOraAUAzNMp2PeABQKxW7yag8gFA833OAHGzCUA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"YCH9OG6v0z8MIac4uwLrP0bXzCErD+8/ina7vGRK8D+SOcEfmVbxP+dtGLDK5/E/vIlN5qiA8j+u8/xLGwL0P3DEwhEsR/Q/gqdTy+oK9T8IyTOlHUf1Py4wy+63ZvU/tV1k8WCJ9T8sZfB5lML1P9w3IspFsfY/OBYkZBhe9z+Y4K4NVab3P44DIWUoZvg/qRnJ/yUg+T8w9YI92Zr5PxMt2yfqrvk/mt9C5SsU+j+guXYvWJj6P9bAfR67tPo/Q+VFP4q6+j/oUghwfPX7PzRQs8XAavw/r5ov8Lmx/D8oWVnyQAf9P4ED/UvHI/0/7CZGzZia/T+orweMMrn9P5YfZou6Hv4/El5ygi19/j82J98ZVX/+Px/fLtwjo/4/UGioFWi+/j/XyACg48z+Pz9IxaY7Jv8/0SXd83wq/z/cY5ACqTv/P0pQsyIlof8/W7bFdGLS/z+kP2L7M+L/P1IhuB2e6f8/4Iy0cTgQAECmO0HcCGwAQBCW+3rweQBASP3NycGgAECWEJ7rQ6cAQAFbGVrhwgBA/Ov5bdDGAEC5UF5xRwYBQP+ybSyUTAFA1RnfxP9RAUAnyR56J4MBQN6H9X88ogFAjyP19Xj4AUD7T0CB5f0BQFKtkLiBHgJAOVaafVREAkDOqBp/xFICQIqFnCeQfgJApbmME7e+AkBtRV20ENMCQC0bk3mD7gJAQBC6c+EjA0ALsW30fy8DQOyCGGN3MwNAck45VX9XA0AvZixof2IDQCk63drbagNAMyrVpjlxA0AIfm5JIpQDQJoz7op6mQNANrTFl8LzA0DRX67XRR8EQFOQAiTwZwRAEkCuev9zBEB68JjhpYIEQJV1JbEEoQRAVzlWJxuqBECWx/NpD8sEQBoGFzYf2gRAf8/tZURlBUAE6+d8KXgFQB9SB2qThwVAQnht63+OBUDie0zW5rUFQDeHpUDXxwVAAEAKiX8nBkAreZjZw+MGQBGo2vucNQdA0Fd4iypXB0Bgx+s91UMIQA4tcnzziQhAZCg5FnXtCEAaGmW7HvAIQFardxNQ+QhA+j5ngLjZDEA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5293"},"selection_policy":{"id":"5292"}},"id":"5277","type":"ColumnDataSource"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5262","type":"PolyAnnotation"},{"attributes":{"formatter":{"id":"5285"},"ticker":{"id":"5250"}},"id":"5249","type":"LinearAxis"},{"attributes":{},"id":"5292","type":"UnionRenderers"},{"attributes":{"axis":{"id":"5249"},"dimension":1,"ticker":null},"id":"5252","type":"Grid"},{"attributes":{"data_source":{"id":"5277"},"glyph":{"id":"5278"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5279"},"selection_glyph":null,"view":{"id":"5281"}},"id":"5280","type":"GlyphRenderer"},{"attributes":{},"id":"5293","type":"Selection"},{"attributes":{"formatter":{"id":"5287"},"ticker":{"id":"5246"}},"id":"5245","type":"LinearAxis"},{"attributes":{"overlay":{"id":"5262"}},"id":"5257","type":"LassoSelectTool"},{"attributes":{},"id":"5253","type":"ResetTool"},{"attributes":{"axis":{"id":"5245"},"ticker":null},"id":"5248","type":"Grid"},{"attributes":{"overlay":{"id":"5261"}},"id":"5255","type":"BoxZoomTool"},{"attributes":{},"id":"5250","type":"BasicTicker"},{"attributes":{},"id":"5254","type":"PanTool"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5291"},"selection_policy":{"id":"5290"}},"id":"5272","type":"ColumnDataSource"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5278","type":"Line"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5273","type":"Patch"},{"attributes":{},"id":"5259","type":"SaveTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5261","type":"BoxAnnotation"},{"attributes":{},"id":"5243","type":"LinearScale"},{"attributes":{"callback":null},"id":"5260","type":"HoverTool"},{"attributes":{"text":""},"id":"5283","type":"Title"},{"attributes":{},"id":"5258","type":"UndoTool"},{"attributes":{},"id":"5241","type":"LinearScale"},{"attributes":{"below":[{"id":"5245"}],"center":[{"id":"5248"},{"id":"5252"}],"left":[{"id":"5249"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5275"},{"id":"5280"}],"title":{"id":"5283"},"toolbar":{"id":"5263"},"toolbar_location":"above","x_range":{"id":"5237"},"x_scale":{"id":"5241"},"y_range":{"id":"5239"},"y_scale":{"id":"5243"}},"id":"5236","subtype":"Figure","type":"Plot"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5274","type":"Patch"},{"attributes":{"source":{"id":"5272"}},"id":"5276","type":"CDSView"},{"attributes":{},"id":"5290","type":"UnionRenderers"},{"attributes":{},"id":"5291","type":"Selection"},{"attributes":{"data_source":{"id":"5272"},"glyph":{"id":"5273"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5274"},"selection_glyph":null,"view":{"id":"5276"}},"id":"5275","type":"GlyphRenderer"},{"attributes":{},"id":"5285","type":"BasicTickFormatter"},{"attributes":{},"id":"5237","type":"DataRange1d"}],"root_ids":["5236"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"96d93505-213d-47ae-ab30-96bde07684a9","root_ids":["5236"],"roots":{"5236":"27e2e92c-12c7-4e9b-bb58-875e9804fc1c"}}];
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