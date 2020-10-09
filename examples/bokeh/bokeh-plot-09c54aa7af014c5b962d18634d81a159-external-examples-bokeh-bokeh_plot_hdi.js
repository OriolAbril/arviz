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
    
      
      
    
      var element = document.getElementById("c19efebe-b16b-4043-a459-82ff9cb6fb2c");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'c19efebe-b16b-4043-a459-82ff9cb6fb2c' but no matching script tag was found.")
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
                    
                  var docs_json = '{"9d5ad267-6cf7-4d9c-a736-2e164acb3889":{"roots":{"references":[{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5273","type":"Patch"},{"attributes":{"text":""},"id":"5283","type":"Title"},{"attributes":{"overlay":{"id":"5262"}},"id":"5257","type":"LassoSelectTool"},{"attributes":{},"id":"5254","type":"PanTool"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5279","type":"Line"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5262","type":"PolyAnnotation"},{"attributes":{},"id":"5246","type":"BasicTicker"},{"attributes":{"source":{"id":"5277"}},"id":"5281","type":"CDSView"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5278","type":"Line"},{"attributes":{},"id":"5284","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5261","type":"BoxAnnotation"},{"attributes":{"data_source":{"id":"5272"},"glyph":{"id":"5273"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5274"},"selection_glyph":null,"view":{"id":"5276"}},"id":"5275","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"5261"}},"id":"5255","type":"BoxZoomTool"},{"attributes":{},"id":"5290","type":"Selection"},{"attributes":{},"id":"5259","type":"SaveTool"},{"attributes":{},"id":"5291","type":"UnionRenderers"},{"attributes":{"below":[{"id":"5245"}],"center":[{"id":"5248"},{"id":"5252"}],"left":[{"id":"5249"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5275"},{"id":"5280"}],"title":{"id":"5283"},"toolbar":{"id":"5263"},"toolbar_location":"above","x_range":{"id":"5237"},"x_scale":{"id":"5241"},"y_range":{"id":"5239"},"y_scale":{"id":"5243"}},"id":"5236","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"5239","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5253"},{"id":"5254"},{"id":"5255"},{"id":"5256"},{"id":"5257"},{"id":"5258"},{"id":"5259"},{"id":"5260"}]},"id":"5263","type":"Toolbar"},{"attributes":{"axis":{"id":"5249"},"dimension":1,"ticker":null},"id":"5252","type":"Grid"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5290"},"selection_policy":{"id":"5291"}},"id":"5272","type":"ColumnDataSource"},{"attributes":{},"id":"5237","type":"DataRange1d"},{"attributes":{},"id":"5243","type":"LinearScale"},{"attributes":{"callback":null},"id":"5260","type":"HoverTool"},{"attributes":{"source":{"id":"5272"}},"id":"5276","type":"CDSView"},{"attributes":{"data":{"x":{"__ndarray__":"u9BLN7sK+785W9Sk4mL6v3R61q7RQfq/xTwjVdi7979n2xGG4eL2vxU7eLK5fva/63gEdPpH9r+THtzV5cvyv80rnm/9yPK/KdLRauuK8r+3g8UvnYjyv97EvTMSiPK/Rx0hKrIw8r8ND7T2J/vwvxgh87mpofC/OVOJydhB77/UgZT2Bdbuvzjwxjegr+6/RaQbxDtr6r8LuX4s2Prpv0adZ2e3o+a/AT78Sh125b9HCkEplGXlvwEDZVDDCOS/d+LMX+SL4r/euhNyjS3gv+ZV7sF93d+/5UHXxEDl3b+sWzjdN53dv1q+pvEQB9y/jsOSQ4pr27/TYqv3BPHZv4eWpkcYMdm/3EyysYna17/dSnPBS9PVv4Qyd+PiutC/u/ljHT6x0L/fRYKBl1DQv+WMX4IfNMy/pnkCQjc8y78rd9f8oLXGv/elB9DOlsW/peCJ501Exb+0HvvM4nrDv1ZMl3/TMMC/++2PkrUjv792HSE42VawvxQ+wMo2TbC/3B1Iyy7thD+UOnUaiq2TP+kLVZb9UrE/tazuwUDBuD/qAAVuFMC5P3NUIqrDML4/E0mUzMnxwD/phHQLBXLPP5joFEUYXNA/turY2ibr0T+6aDeVeT7SPzH+7p6PRNI/aTuIeMcs1D/V/q36+J7UP9anK8yp6dQ/PezmxiHz1D+o4kL04I/XPxE628ZQldo/nazcqiMD4D9+Gqqa+wrgP0C1pF6IguE/yvv3mpyo4j8luWY093rkP1rPqVZhueU/lY+umSlL5z/A4vOHSY/rP+0ZCxh9kus/eEn/5K7o7T8AhyseH/ntP0o/IugtOO4/lZIasZN67j9n8JXIMAbvP8ZqtDJO1u8/BNVLWt4A8D/qsMs02xbwPy7QD7gGL/A/cfj6tvpI8D8tSucvUYvyP2pwRLChEvM/L2q/gNde8z+xEWtlmW3zP+yOD1ETmfM/U0SpG/Hd9D9wBfU7Xa/1P92qmrl8sPY/TkSzVpo69z9ygdGCkYb3PwQ/rVW4nvg/Spzl/t78+T+aBXcW5ZX/P5tfaSMchABAV6yLzr0KAUA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"ohdaZKJ68j9k0pWtjs7yP8bClCgX3/I/nmFu1RMi9D9MEvc8j470P3biwyajwPQ/isP9xQLc9D+28BEVDZr2PxrqMEiBm/Y/7BaXSoq69j8kPh1osbv2P5EdIeb2u/Y/XHHv6qbn9j96+KUEbIL3P3RvBiMrr/c/MqudzYkv+D+L31qCfkr4P/JDDvIXVPg/7xb5DjFl+T+9UeD0SYH5P64YJiYSV/o/gPBArXii+j9uva/1mqb6P0C/5ivP/fo/YscM6AZd+z9IEXujnPT7P0M1wkdQBPw/wxdl51dD/D+K9FgEWUz8PzUoy+Edf/w/jqeNt46S/D+mkwph38H8Py8tC/fc2fw/ZLbJya4E/T+kltGHlkX9P7AZkaOj6P0/yYBTPNjp/T9Et88P7fX9PzIH2ge+PP4/ZtjfizxM/j+NiDLwpZT+P6GF/xKTpv4/9mGHIbur/j8VTjDTUcj+PzuLBsjy/P4/kIBrU+IG/z8U9z42SX3/Pw/+qUmWff8/D6Rll3YKAEA7dRqKrRMAQDBUWfZLRQBAs7oHAwVjAEAEFLhRAGcAQFKJqA7DeABASaJkTo6HAEAnpFsokPsAQIpOUYTBBQFAq46tbbIeAUCMdlOZ5yMBQOPv7vlIJAFAt4OId8xCAUDt36qP70kBQH26wpyaTgFAxG5uHDJPAUAqLkQP/ngBQKGzbQxVqQFAlJVbdWQAAkBQQ1VzXwECQKiW1AtRMAJAef9ekxNVAkAl14zmXo8CQOs51SostwJA89E1M2XpAkBYfP4w6XEDQD5jAaNPcgNAL+mf3BW9A0DgcMXjI78DQOlHBL0FxwNAU1IjdlLPA0ANvhIZxuADQFmNVsbJ+gNAQfWSljcABEA67DLNtgUEQAz0A67BCwRAHL6+rT4SBECL0vlL1KIEQBocEWyoxARAjNov4LXXBEBsxFpZZtsEQLvjQ9RE5gRAFVHqRnw3BUBcQf1O12sFQLeqZi4frAVAFNGslabOBUBcYLRgpOEFQMFPaxWuJwZAEme5vzd/BkBmwZ1FeeUHQM6vtBEOQghALNZF516FCEA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5292"},"selection_policy":{"id":"5293"}},"id":"5277","type":"ColumnDataSource"},{"attributes":{},"id":"5292","type":"Selection"},{"attributes":{},"id":"5293","type":"UnionRenderers"},{"attributes":{},"id":"5286","type":"BasicTickFormatter"},{"attributes":{},"id":"5250","type":"BasicTicker"},{"attributes":{"data_source":{"id":"5277"},"glyph":{"id":"5278"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5279"},"selection_glyph":null,"view":{"id":"5281"}},"id":"5280","type":"GlyphRenderer"},{"attributes":{"formatter":{"id":"5286"},"ticker":{"id":"5246"}},"id":"5245","type":"LinearAxis"},{"attributes":{},"id":"5241","type":"LinearScale"},{"attributes":{"axis":{"id":"5245"},"ticker":null},"id":"5248","type":"Grid"},{"attributes":{"formatter":{"id":"5284"},"ticker":{"id":"5250"}},"id":"5249","type":"LinearAxis"},{"attributes":{},"id":"5253","type":"ResetTool"},{"attributes":{},"id":"5256","type":"WheelZoomTool"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5274","type":"Patch"},{"attributes":{},"id":"5258","type":"UndoTool"}],"root_ids":["5236"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"9d5ad267-6cf7-4d9c-a736-2e164acb3889","root_ids":["5236"],"roots":{"5236":"c19efebe-b16b-4043-a459-82ff9cb6fb2c"}}];
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