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
    
      
      
    
      var element = document.getElementById("8ffccebd-bed5-4c2c-a275-fa5e30f4b75b");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '8ffccebd-bed5-4c2c-a275-fa5e30f4b75b' but no matching script tag was found.")
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
                    
                  var docs_json = '{"9c174829-bbf1-4dc7-9487-21342663f459":{"roots":{"references":[{"attributes":{"axis":{"id":"5337"},"ticker":null},"id":"5340","type":"Grid"},{"attributes":{"formatter":{"id":"5378"},"ticker":{"id":"5338"}},"id":"5337","type":"LinearAxis"},{"attributes":{"source":{"id":"5364"}},"id":"5368","type":"CDSView"},{"attributes":{},"id":"5350","type":"UndoTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5345"},{"id":"5346"},{"id":"5347"},{"id":"5348"},{"id":"5349"},{"id":"5350"},{"id":"5351"},{"id":"5352"}]},"id":"5355","type":"Toolbar"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5382"},"selection_policy":{"id":"5383"}},"id":"5364","type":"ColumnDataSource"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5365","type":"Patch"},{"attributes":{"callback":null},"id":"5352","type":"HoverTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5353","type":"BoxAnnotation"},{"attributes":{},"id":"5382","type":"Selection"},{"attributes":{"axis":{"id":"5341"},"dimension":1,"ticker":null},"id":"5344","type":"Grid"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5354","type":"PolyAnnotation"},{"attributes":{},"id":"5333","type":"LinearScale"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5366","type":"Patch"},{"attributes":{"below":[{"id":"5337"}],"center":[{"id":"5340"},{"id":"5344"}],"left":[{"id":"5341"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5367"},{"id":"5372"}],"title":{"id":"5374"},"toolbar":{"id":"5355"},"toolbar_location":"above","x_range":{"id":"5329"},"x_scale":{"id":"5333"},"y_range":{"id":"5331"},"y_scale":{"id":"5335"}},"id":"5328","subtype":"Figure","type":"Plot"},{"attributes":{"overlay":{"id":"5353"}},"id":"5347","type":"BoxZoomTool"},{"attributes":{},"id":"5346","type":"PanTool"},{"attributes":{},"id":"5348","type":"WheelZoomTool"},{"attributes":{},"id":"5329","type":"DataRange1d"},{"attributes":{},"id":"5378","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"5354"}},"id":"5349","type":"LassoSelectTool"},{"attributes":{},"id":"5345","type":"ResetTool"},{"attributes":{},"id":"5376","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"5369"},"glyph":{"id":"5370"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5371"},"selection_glyph":null,"view":{"id":"5373"}},"id":"5372","type":"GlyphRenderer"},{"attributes":{},"id":"5342","type":"BasicTicker"},{"attributes":{},"id":"5385","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5371","type":"Line"},{"attributes":{"data":{"x":{"__ndarray__":"ih7xvJfwBMAZ6vxOMjsEwNeLuD9a8gPAoLEeepV8A8C7TXKDm60BwHEXDVxiiQHAXPihWf+AAcCr+s8gCVX/v4F9b7LNcPy/ljRmvWZo+7/OUBvI1nL5v59YrPUiHPm/Q4/xENku978NJMXoItf2v2zLM+q1nvW/qdPw9R519b8zBNpIl73wv8KsM1MJHPC/fOb1XG+t7r/pIE5Qp6nuv430axyFpu6/0j9B79OF7b+XRHU5Nprsv/W4NzE6Ney/CGeciBEV7L83pcC/HsTpv5QFdFfDhOm/pLrlN7856b924zLu1EbnvwkGuz4ibua/n4lT96Sg5b+4oUhOM3flvzTYht+dhuO/38bgnwEA479Et/mDHETiv8ck4Xqh5uG/igDEYlxp3b+5B3LtZgHcv4PTPWflxtu/kL3Goup227/niXXUuUXbv2yR3heSONu/VQWxiWWT2r/fSJ0H+BXUv0pBftx7Xc6/9/VApnRfzL/LHFG2O0fLv2Y/qEnc3Mq/Z/TtSPKKyr8eej8xzybGv3c29YQ3Asa/olvXlQX8xb+nkUQ3uzvDv31xYo/TUbW/eg4dP73tsD9dHLC92DyxP23GVL0Ha7Q/QOkdGlFpuj8h41CWSc6+P72LXGakT8U/aUqfYidF0j86xrDNca/UP5dg2OazhNk/552bDbCN2j81BtF3GKzeP+tGUBE89eA/zAH3dy8v4T8crMGVp/HiP8Gis4fc7uM/rt1o+PIx5T+V344ub4nlPyKQp2WBmeU/XybVLPJx5j9xwSqF3zLnP40lYSebn+g/pGeKZaXr6T/sNMf4wQ3qP1iEXHctHOw/xlHqZ5su7T9Y9YKf+VjtPwQBmqxJrO0/83uVWZAD7z/Y9Sqr4TnvP/6Cr6IKtu8/unJqz56t8D8TjusZnuDwP/jkmePVhPE/EWXuWyUV8z/rkoShTBnzP+5+XB4HP/M/2iKJ4Y3I8z9GpHZsMhv0P6zk3LUt9fQ/rgRY7YNC9T+BM6TRlZD1P9EBbRw4jfY/nG0B/Elt+T9kHje0ehX6P7KoEd28bP0/ckimGr+yAUA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"7MIdhtAe5j/OKwZim4nnP1LojoBLG+g/wJzCC9UG6T+KZBv5yKTsPx7R5Uc77ew/SA+8TAH+7D+qAphve1XwP0BByCaZx/E/teVMocxL8j+ZV/KblEbzP7DTKYXucfM/XjiHd5No9D/6bZ2LbpT0P0oa5gqlMPU/LJYHhXBF9T/m/ZJbNKH3P58pZlb78fc/YYbCKKRU+D/Gd+wrllX4P90C5bheVvg/DLAvBIue+D/arqJxctn4P8MRsnOx8vg/PubYnbv6+D+y1g9Q+I75P5v+IirPnvk/V5EGMpCx+T8iR3PESi76P34+UXB3ZPo/mB0rwtaX+j+S120sM6L6P/NJHohYHvs/SM4HmP8/+z8vkgHf+G77P862R6FXhvs/73+nc9RS/D8Jv1Ei03/8P5BFGFMjh/w/TiinqyKR/D/DTnHFSJf8P9ItBL3tmPw/Vd/JTpOt/D/kVgz/QH39P+sbOEIoGv4/ofCbtQg6/j8z7ppEjEv+Pwp8ZTsyUv4/uiBx21BX/j9eCOwMk53+P5mssIfcn/4/Roqipj+g/j/mtotMRMz+P3TshGNxVf8/OnT89LZDAEBxwPZi80QAQBpT9R6sUQBApXdoRKVpAECNQ1kmOXsAQF7kMiN9qgBAp/QpdlIkAUBkDNsc90oBQAmGbT5LmAFA3rnZANuoAUBjEH2HweoBQN0IKoKnHgJAOuD+7uUlAkCENbjyNF4CQFh09pDbfQJAthsNXz6mAkDz29HlLbECQATytCwwswJAzKSaRT7OAkAuWKXwW+YCQLIk7GTzEwNA9EyxrHQ9A0Ce5hg/uEEDQIuQ666FgwNAOUr9bNOlA0CrXvAzH6sDQCBAkzWJtQNAfq8yC3LgA0C7XmU1POcDQGDwVVTB9gNArpzas2crBECF43qGJzgEQD555ng1YQRARJn7VknFBEC7JGEoU8YEQLwfl8fBzwRAtkhieCPyBEASqR2bzAYFQCs5d21LPQVALAFW+6BQBUDgDGl0JWQFQHRAGwdOowVAZ1sAf1JbBkCZxw2tXoUGQCxqRDcvWwdAOSRTjV/ZCEA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5384"},"selection_policy":{"id":"5385"}},"id":"5369","type":"ColumnDataSource"},{"attributes":{},"id":"5331","type":"DataRange1d"},{"attributes":{},"id":"5383","type":"UnionRenderers"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5370","type":"Line"},{"attributes":{"source":{"id":"5369"}},"id":"5373","type":"CDSView"},{"attributes":{},"id":"5351","type":"SaveTool"},{"attributes":{"data_source":{"id":"5364"},"glyph":{"id":"5365"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5366"},"selection_glyph":null,"view":{"id":"5368"}},"id":"5367","type":"GlyphRenderer"},{"attributes":{},"id":"5335","type":"LinearScale"},{"attributes":{"text":""},"id":"5374","type":"Title"},{"attributes":{},"id":"5384","type":"Selection"},{"attributes":{"formatter":{"id":"5376"},"ticker":{"id":"5342"}},"id":"5341","type":"LinearAxis"},{"attributes":{},"id":"5338","type":"BasicTicker"}],"root_ids":["5328"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"9c174829-bbf1-4dc7-9487-21342663f459","root_ids":["5328"],"roots":{"5328":"8ffccebd-bed5-4c2c-a275-fa5e30f4b75b"}}];
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